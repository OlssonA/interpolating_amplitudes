module     p0_ubaru_httbar_abbrevd67h2
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh2
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
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb67(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb67(2)=NC**(-1)
      abb67(3)=spak2l3**(-1)
      abb67(4)=spbl3k2**(-1)
      abb67(5)=spbl5k2**(-1)
      abb67(6)=sqrt(mT**2)
      abb67(7)=spbl4k2**(-1)
      abb67(8)=i_*e*gHT*abb67(1)*TR**2*gs**4
      abb67(9)=abb67(8)*mT**2
      abb67(10)=abb67(9)*abb67(2)
      abb67(11)=spbk2k1*spak2l4
      abb67(12)=abb67(10)*abb67(11)
      abb67(13)=abb67(5)*abb67(12)
      abb67(14)=abb67(8)*spak2l4
      abb67(15)=abb67(2)*mT
      abb67(16)=abb67(14)*abb67(15)
      abb67(17)=abb67(6)*abb67(5)
      abb67(18)=abb67(16)*abb67(17)
      abb67(19)=abb67(18)*spbk2k1
      abb67(13)=abb67(13)+abb67(19)
      abb67(19)=2.0_ki*c2
      abb67(13)=abb67(13)*abb67(19)
      abb67(14)=abb67(14)*abb67(2)
      abb67(20)=abb67(14)*abb67(19)
      abb67(21)=abb67(20)*spbk2k1
      abb67(22)=abb67(2)**2
      abb67(22)=abb67(22)+1.0_ki
      abb67(23)=abb67(8)*abb67(22)
      abb67(11)=abb67(11)*c1
      abb67(24)=abb67(23)*abb67(11)
      abb67(21)=abb67(21)-abb67(24)
      abb67(25)=abb67(3)*abb67(4)*spak2l5*mH**2
      abb67(21)=abb67(21)*abb67(25)
      abb67(26)=mT*abb67(23)
      abb67(27)=abb67(26)*abb67(17)
      abb67(9)=abb67(9)*abb67(22)
      abb67(22)=abb67(9)*abb67(5)
      abb67(28)=abb67(27)+abb67(22)
      abb67(28)=abb67(11)*abb67(28)
      abb67(29)=c1*spak2l4
      abb67(30)=abb67(23)*abb67(29)
      abb67(20)=abb67(30)-abb67(20)
      abb67(31)=spbl3k1*spal3l5
      abb67(20)=abb67(20)*abb67(31)
      abb67(13)=-abb67(21)-abb67(13)+abb67(28)+abb67(20)
      abb67(13)=4.0_ki*abb67(13)
      abb67(20)=spbk2k1*mT**4
      abb67(21)=-abb67(7)*abb67(8)*abb67(2)*abb67(20)
      abb67(12)=abb67(12)+abb67(21)
      abb67(12)=abb67(5)*abb67(12)
      abb67(21)=abb67(6)*abb67(7)
      abb67(28)=abb67(10)*abb67(21)
      abb67(28)=abb67(16)+abb67(28)
      abb67(17)=spbk2k1*abb67(28)*abb67(17)
      abb67(12)=abb67(12)+abb67(17)
      abb67(12)=abb67(12)*abb67(19)
      abb67(17)=abb67(21)*c1
      abb67(28)=abb67(17)*abb67(26)
      abb67(32)=abb67(7)*c1
      abb67(33)=abb67(9)*abb67(32)
      abb67(34)=abb67(28)-abb67(33)
      abb67(35)=-spbk2k1*abb67(34)
      abb67(8)=abb67(8)*abb67(21)*abb67(15)
      abb67(10)=abb67(10)*abb67(7)
      abb67(15)=abb67(8)-abb67(10)+abb67(14)
      abb67(36)=abb67(19)*spbk2k1*abb67(15)
      abb67(24)=abb67(36)-abb67(24)+abb67(35)
      abb67(24)=abb67(24)*abb67(25)
      abb67(15)=abb67(15)*abb67(19)
      abb67(15)=abb67(15)-abb67(30)-abb67(34)
      abb67(15)=abb67(15)*abb67(31)
      abb67(20)=abb67(32)*abb67(23)*abb67(20)
      abb67(23)=-abb67(9)*abb67(11)
      abb67(20)=abb67(23)+abb67(20)
      abb67(20)=abb67(5)*abb67(20)
      abb67(11)=-abb67(5)*abb67(26)*abb67(11)
      abb67(17)=-spbk2k1*abb67(17)*abb67(22)
      abb67(11)=abb67(11)+abb67(17)
      abb67(11)=abb67(6)*abb67(11)
      abb67(11)=abb67(15)+abb67(24)+abb67(12)+abb67(20)+abb67(11)
      abb67(11)=4.0_ki*abb67(11)
      abb67(12)=abb67(29)*abb67(7)
      abb67(9)=abb67(9)*abb67(12)
      abb67(15)=abb67(26)*abb67(29)
      abb67(17)=abb67(21)*abb67(15)
      abb67(20)=-abb67(21)*abb67(16)
      abb67(21)=-spak2l4*abb67(10)
      abb67(20)=abb67(21)+abb67(20)
      abb67(20)=abb67(20)*abb67(19)
      abb67(9)=abb67(20)+abb67(9)+abb67(17)
      abb67(9)=2.0_ki*spbl4k1*abb67(9)
      abb67(17)=-abb67(6)*abb67(30)
      abb67(14)=abb67(6)*abb67(14)
      abb67(14)=abb67(16)+abb67(14)
      abb67(14)=abb67(14)*abb67(19)
      abb67(14)=abb67(14)-abb67(15)+abb67(17)
      abb67(14)=2.0_ki*abb67(6)*abb67(14)
      abb67(12)=abb67(22)*abb67(12)
      abb67(15)=abb67(5)*abb67(10)*abb67(19)
      abb67(16)=-spak2l4*abb67(15)
      abb67(12)=abb67(12)+abb67(16)
      abb67(16)=2.0_ki*spbl3k2
      abb67(12)=spbl4k1*abb67(12)*abb67(16)
      abb67(17)=-abb67(29)*abb67(27)
      abb67(18)=abb67(19)*abb67(18)
      abb67(17)=abb67(17)+abb67(18)
      abb67(17)=abb67(17)*abb67(16)
      abb67(8)=-abb67(10)-abb67(8)
      abb67(8)=abb67(8)*abb67(19)
      abb67(8)=abb67(8)+abb67(33)+abb67(28)
      abb67(8)=2.0_ki*abb67(8)
      abb67(10)=abb67(32)*abb67(22)
      abb67(10)=abb67(10)-abb67(15)
      abb67(10)=abb67(10)*abb67(16)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd67h2
