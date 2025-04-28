module     p0_ubaru_httbar_abbrevd67h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(36), public :: abb67
   complex(ki), public :: R2d67
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb67(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb67(2)=NC**(-1)
      abb67(3)=spak2l3**(-1)
      abb67(4)=spbl3k2**(-1)
      abb67(5)=spak2l5**(-1)
      abb67(6)=sqrt(mT**2)
      abb67(7)=spak2l4**(-1)
      abb67(8)=i_*e*gHT*abb67(1)*TR**2*gs**4
      abb67(9)=abb67(8)*mT**2
      abb67(10)=abb67(9)*abb67(2)
      abb67(11)=spak1k2*spbl4k2
      abb67(12)=abb67(10)*abb67(11)
      abb67(13)=abb67(5)*abb67(12)
      abb67(14)=abb67(8)*spbl4k2
      abb67(15)=abb67(2)*mT
      abb67(16)=abb67(14)*abb67(15)
      abb67(17)=abb67(6)*abb67(5)
      abb67(18)=abb67(16)*abb67(17)
      abb67(19)=abb67(18)*spak1k2
      abb67(13)=abb67(13)+abb67(19)
      abb67(19)=2.0_ki*c2
      abb67(13)=abb67(13)*abb67(19)
      abb67(14)=abb67(14)*abb67(2)
      abb67(20)=abb67(14)*abb67(19)
      abb67(21)=abb67(20)*spak1k2
      abb67(22)=abb67(2)**2
      abb67(22)=abb67(22)+1.0_ki
      abb67(23)=abb67(8)*abb67(22)
      abb67(11)=abb67(11)*c1
      abb67(24)=abb67(23)*abb67(11)
      abb67(21)=abb67(21)-abb67(24)
      abb67(25)=abb67(3)*abb67(4)*spbl5k2*mH**2
      abb67(21)=abb67(21)*abb67(25)
      abb67(26)=mT*abb67(23)
      abb67(27)=abb67(26)*abb67(17)
      abb67(9)=abb67(9)*abb67(22)
      abb67(22)=abb67(9)*abb67(5)
      abb67(28)=abb67(27)+abb67(22)
      abb67(28)=abb67(11)*abb67(28)
      abb67(29)=c1*spbl4k2
      abb67(30)=abb67(23)*abb67(29)
      abb67(20)=abb67(30)-abb67(20)
      abb67(31)=spak1l3*spbl5l3
      abb67(20)=abb67(20)*abb67(31)
      abb67(13)=-abb67(21)-abb67(13)+abb67(28)+abb67(20)
      abb67(13)=4.0_ki*abb67(13)
      abb67(20)=spak1k2*mT**4
      abb67(21)=abb67(7)*abb67(8)*abb67(2)*abb67(20)
      abb67(12)=-abb67(12)+abb67(21)
      abb67(12)=abb67(5)*abb67(12)
      abb67(21)=abb67(6)*abb67(7)
      abb67(28)=-abb67(10)*abb67(21)
      abb67(28)=-abb67(16)+abb67(28)
      abb67(17)=spak1k2*abb67(28)*abb67(17)
      abb67(12)=abb67(12)+abb67(17)
      abb67(12)=abb67(12)*abb67(19)
      abb67(17)=abb67(21)*c1
      abb67(28)=abb67(17)*abb67(26)
      abb67(32)=abb67(7)*c1
      abb67(33)=abb67(9)*abb67(32)
      abb67(34)=abb67(28)-abb67(33)
      abb67(35)=spak1k2*abb67(34)
      abb67(8)=abb67(8)*abb67(21)*abb67(15)
      abb67(10)=abb67(10)*abb67(7)
      abb67(15)=abb67(8)-abb67(10)+abb67(14)
      abb67(36)=-abb67(19)*spak1k2*abb67(15)
      abb67(24)=abb67(36)+abb67(24)+abb67(35)
      abb67(24)=abb67(24)*abb67(25)
      abb67(15)=-abb67(15)*abb67(19)
      abb67(15)=abb67(15)+abb67(30)+abb67(34)
      abb67(15)=abb67(15)*abb67(31)
      abb67(20)=-abb67(32)*abb67(23)*abb67(20)
      abb67(23)=abb67(9)*abb67(11)
      abb67(20)=abb67(23)+abb67(20)
      abb67(20)=abb67(5)*abb67(20)
      abb67(11)=abb67(5)*abb67(26)*abb67(11)
      abb67(17)=spak1k2*abb67(17)*abb67(22)
      abb67(11)=abb67(11)+abb67(17)
      abb67(11)=abb67(6)*abb67(11)
      abb67(11)=abb67(15)+abb67(24)+abb67(12)+abb67(20)+abb67(11)
      abb67(11)=4.0_ki*abb67(11)
      abb67(12)=abb67(29)*abb67(7)
      abb67(9)=-abb67(9)*abb67(12)
      abb67(15)=abb67(26)*abb67(29)
      abb67(17)=-abb67(21)*abb67(15)
      abb67(20)=abb67(21)*abb67(16)
      abb67(21)=spbl4k2*abb67(10)
      abb67(20)=abb67(21)+abb67(20)
      abb67(20)=abb67(20)*abb67(19)
      abb67(9)=abb67(20)+abb67(9)+abb67(17)
      abb67(9)=2.0_ki*spak1l4*abb67(9)
      abb67(12)=-abb67(22)*abb67(12)
      abb67(17)=abb67(5)*abb67(10)*abb67(19)
      abb67(20)=spbl4k2*abb67(17)
      abb67(12)=abb67(12)+abb67(20)
      abb67(20)=2.0_ki*spak2l3
      abb67(12)=spak1l4*abb67(12)*abb67(20)
      abb67(21)=abb67(6)*abb67(30)
      abb67(14)=-abb67(6)*abb67(14)
      abb67(14)=-abb67(16)+abb67(14)
      abb67(14)=abb67(14)*abb67(19)
      abb67(14)=abb67(14)+abb67(15)+abb67(21)
      abb67(14)=2.0_ki*abb67(6)*abb67(14)
      abb67(15)=abb67(29)*abb67(27)
      abb67(16)=-abb67(19)*abb67(18)
      abb67(15)=abb67(15)+abb67(16)
      abb67(15)=abb67(15)*abb67(20)
      abb67(8)=abb67(10)+abb67(8)
      abb67(8)=abb67(8)*abb67(19)
      abb67(8)=abb67(8)-abb67(33)-abb67(28)
      abb67(8)=2.0_ki*abb67(8)
      abb67(10)=-abb67(32)*abb67(22)
      abb67(10)=abb67(10)+abb67(17)
      abb67(10)=abb67(10)*abb67(20)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd67h13_qp
