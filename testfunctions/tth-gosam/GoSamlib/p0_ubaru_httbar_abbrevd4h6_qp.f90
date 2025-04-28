module     p0_ubaru_httbar_abbrevd4h6_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh6_qp
   implicit none
   private
   complex(ki), dimension(37), public :: abb4
   complex(ki), public :: R2d4
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
      abb4(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb4(2)=es12**(-1)
      abb4(3)=spbl5k2**(-1)
      abb4(4)=sqrt(mT**2)
      abb4(5)=spak2l3**(-1)
      abb4(6)=spbl3k2**(-1)
      abb4(7)=spak2l4**(-1)
      abb4(8)=spbl4k2**(-1)
      abb4(9)=NC*c2
      abb4(9)=abb4(9)-c1
      abb4(9)=abb4(9)*i_*e*gHT*abb4(1)*TR**2*gs**4
      abb4(10)=-abb4(2)*abb4(9)
      abb4(11)=spbl4k1*mT
      abb4(12)=-abb4(11)*abb4(10)
      abb4(13)=-spbl4k1*abb4(10)
      abb4(14)=abb4(13)*abb4(4)
      abb4(15)=abb4(14)+abb4(12)
      abb4(16)=abb4(15)*spak2l5
      abb4(17)=spbl3k2*abb4(3)
      abb4(18)=abb4(17)*spak2l3
      abb4(19)=abb4(18)*abb4(12)
      abb4(16)=abb4(16)+abb4(19)
      abb4(19)=abb4(9)*spbl4k1
      abb4(20)=mT**3
      abb4(21)=abb4(20)*abb4(19)
      abb4(22)=mT**2
      abb4(23)=-abb4(22)*abb4(19)
      abb4(24)=-abb4(4)*abb4(23)
      abb4(24)=abb4(24)+abb4(21)
      abb4(25)=abb4(7)*spak2l5
      abb4(24)=abb4(24)*abb4(25)
      abb4(26)=abb4(18)*abb4(7)
      abb4(27)=abb4(21)*abb4(26)
      abb4(24)=abb4(24)+abb4(27)
      abb4(24)=abb4(8)*abb4(24)
      abb4(27)=abb4(10)*abb4(4)
      abb4(28)=spbl4k1**2
      abb4(29)=abb4(28)*abb4(27)
      abb4(30)=-mT*abb4(10)
      abb4(28)=-abb4(28)*abb4(30)
      abb4(29)=abb4(29)+abb4(28)
      abb4(31)=-spak2l5*abb4(29)
      abb4(32)=-abb4(28)*abb4(18)
      abb4(31)=abb4(31)+abb4(32)
      abb4(31)=spak1l4*abb4(31)
      abb4(24)=abb4(24)+abb4(31)
      abb4(31)=spbl4k2*abb4(16)
      abb4(29)=spak1l5*abb4(29)
      abb4(28)=abb4(28)*abb4(17)*spak1l3
      abb4(28)=abb4(28)+abb4(29)+abb4(31)
      abb4(28)=spak2l4*abb4(28)
      abb4(11)=-abb4(9)*abb4(11)
      abb4(29)=3.0_ki*abb4(4)
      abb4(31)=-abb4(11)*abb4(29)
      abb4(23)=abb4(31)-abb4(23)
      abb4(23)=abb4(4)*abb4(23)
      abb4(21)=-2.0_ki*abb4(21)+abb4(23)
      abb4(21)=abb4(3)*abb4(21)
      abb4(19)=abb4(19)*abb4(29)
      abb4(11)=2.0_ki*abb4(11)+abb4(19)
      abb4(19)=mH**2*abb4(6)*abb4(5)
      abb4(23)=abb4(19)*spak2l5
      abb4(11)=abb4(11)*abb4(23)
      abb4(31)=-2.0_ki*abb4(12)+3.0_ki*abb4(14)
      abb4(32)=spbl3k1*spal3l5
      abb4(33)=abb4(32)*spak1k2
      abb4(31)=abb4(31)*abb4(33)
      abb4(11)=abb4(31)+abb4(21)+abb4(11)+abb4(28)+2.0_ki*abb4(24)
      abb4(21)=2.0_ki*abb4(16)
      abb4(13)=-abb4(22)*abb4(13)
      abb4(24)=-abb4(4)*abb4(12)
      abb4(13)=abb4(24)+abb4(13)
      abb4(13)=abb4(3)*abb4(4)*abb4(13)
      abb4(24)=-abb4(14)*abb4(23)
      abb4(13)=abb4(24)+abb4(13)-abb4(16)
      abb4(13)=2.0_ki*abb4(13)
      abb4(24)=4.0_ki*abb4(16)
      abb4(28)=spak2l4*abb4(15)
      abb4(12)=abb4(12)*abb4(17)
      abb4(31)=spak2l4*abb4(12)
      abb4(34)=-abb4(22)*abb4(10)
      abb4(35)=abb4(34)*abb4(4)
      abb4(10)=-abb4(20)*abb4(10)
      abb4(35)=abb4(35)+abb4(10)
      abb4(36)=abb4(35)*abb4(25)
      abb4(26)=abb4(10)*abb4(26)
      abb4(26)=abb4(36)+abb4(26)
      abb4(26)=abb4(8)*abb4(26)
      abb4(36)=abb4(30)*abb4(4)**2
      abb4(36)=abb4(36)-abb4(10)
      abb4(36)=abb4(3)*abb4(36)
      abb4(37)=abb4(27)+abb4(30)
      abb4(23)=-abb4(37)*abb4(23)
      abb4(23)=abb4(26)+abb4(36)+abb4(23)
      abb4(23)=spbk2k1*abb4(23)
      abb4(26)=-abb4(37)*abb4(32)
      abb4(23)=abb4(26)+abb4(23)
      abb4(14)=-spal3l5*abb4(14)
      abb4(26)=-abb4(30)*abb4(29)
      abb4(26)=abb4(26)-abb4(34)
      abb4(26)=abb4(4)*abb4(26)
      abb4(32)=2.0_ki*abb4(10)
      abb4(26)=abb4(32)+abb4(26)
      abb4(26)=spak2l5*abb4(26)
      abb4(29)=-abb4(34)*abb4(29)
      abb4(29)=abb4(32)+abb4(29)
      abb4(29)=abb4(29)*abb4(18)
      abb4(22)=-abb4(4)*abb4(22)
      abb4(20)=abb4(22)-abb4(20)
      abb4(20)=abb4(3)*abb4(9)*abb4(20)
      abb4(22)=2.0_ki*abb4(30)
      abb4(32)=-abb4(22)*abb4(33)
      abb4(20)=abb4(32)+abb4(29)+abb4(26)+2.0_ki*abb4(20)
      abb4(20)=abb4(7)*abb4(20)
      abb4(26)=abb4(4)*abb4(22)
      abb4(26)=abb4(26)+abb4(34)
      abb4(26)=abb4(4)*abb4(26)
      abb4(10)=abb4(26)-abb4(10)
      abb4(10)=abb4(3)*abb4(10)
      abb4(26)=abb4(27)-abb4(30)
      abb4(29)=-spak2l5*abb4(26)
      abb4(18)=abb4(30)*abb4(18)
      abb4(10)=abb4(18)+abb4(29)+abb4(10)
      abb4(10)=spbl4k2*abb4(10)
      abb4(9)=-abb4(25)*mT*abb4(9)
      abb4(18)=abb4(30)+2.0_ki*abb4(27)
      abb4(27)=-spbl4k2*spak2l5*abb4(18)
      abb4(9)=abb4(9)+abb4(27)
      abb4(9)=abb4(9)*abb4(19)
      abb4(18)=spbl4l3*spal3l5*abb4(18)
      abb4(9)=-abb4(18)+abb4(10)+abb4(9)
      abb4(10)=-spak1l5*abb4(15)
      abb4(12)=-spak1l3*abb4(12)
      abb4(9)=abb4(12)+abb4(10)+abb4(20)+2.0_ki*abb4(9)
      abb4(10)=-abb4(7)*abb4(3)*abb4(35)
      abb4(12)=-abb4(30)*abb4(19)*abb4(25)
      abb4(10)=abb4(10)+abb4(12)
      abb4(10)=4.0_ki*abb4(10)
      abb4(12)=-2.0_ki*abb4(26)
      abb4(15)=abb4(17)*abb4(22)
      abb4(17)=-abb4(7)*abb4(22)*spal3l5
      R2d4=-abb4(16)
      rat2 = rat2 + R2d4
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='4' value='", &
          & R2d4, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd4h6_qp
