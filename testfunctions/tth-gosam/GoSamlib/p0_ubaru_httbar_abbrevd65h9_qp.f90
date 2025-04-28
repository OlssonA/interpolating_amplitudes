module     p0_ubaru_httbar_abbrevd65h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
   implicit none
   private
   complex(ki), dimension(30), public :: abb65
   complex(ki), public :: R2d65
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
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=NC**(-1)
      abb65(3)=spbl4k2**(-1)
      abb65(4)=spak2l3**(-1)
      abb65(5)=spbl3k2**(-1)
      abb65(6)=spak2l5**(-1)
      abb65(7)=sqrt(mT**2)
      abb65(8)=spak1l3*abb65(3)*spbl3k2
      abb65(9)=abb65(8)*c1
      abb65(10)=c1*spak1l4
      abb65(9)=abb65(9)+abb65(10)
      abb65(11)=spbl5k2*abb65(9)*abb65(2)
      abb65(12)=c2*spbl5k2
      abb65(13)=2.0_ki*abb65(12)
      abb65(14)=abb65(8)+spak1l4
      abb65(15)=abb65(13)*abb65(14)
      abb65(11)=abb65(11)-abb65(15)
      abb65(11)=abb65(11)*abb65(2)
      abb65(8)=abb65(8)*NC
      abb65(15)=NC*spak1l4
      abb65(8)=abb65(8)+abb65(15)
      abb65(16)=abb65(8)*abb65(12)
      abb65(11)=abb65(11)+abb65(16)
      abb65(16)=abb65(1)*gHT*e*i_*gs**4*TR**2
      abb65(17)=abb65(16)*mT
      abb65(17)=4.0_ki*abb65(17)
      abb65(18)=abb65(11)*abb65(17)
      abb65(19)=abb65(6)*spak1l5
      abb65(20)=abb65(19)*abb65(3)
      abb65(21)=spbl5k2*abb65(7)
      abb65(22)=abb65(20)*abb65(21)
      abb65(23)=abb65(7)*abb65(6)
      abb65(24)=abb65(14)*abb65(23)
      abb65(24)=abb65(24)+abb65(22)
      abb65(25)=2.0_ki*c2
      abb65(24)=abb65(24)*abb65(25)
      abb65(26)=-abb65(9)*abb65(23)
      abb65(27)=-c1*abb65(22)
      abb65(26)=abb65(26)+abb65(27)
      abb65(26)=abb65(2)*abb65(26)
      abb65(24)=abb65(24)+abb65(26)
      abb65(24)=abb65(2)*abb65(24)
      abb65(26)=-abb65(8)*abb65(23)
      abb65(22)=-NC*abb65(22)
      abb65(22)=abb65(26)+abb65(22)
      abb65(22)=c2*abb65(22)
      abb65(20)=abb65(20)*spbl5k2
      abb65(26)=-abb65(6)*abb65(14)
      abb65(26)=abb65(26)+abb65(20)
      abb65(26)=abb65(26)*abb65(25)
      abb65(27)=abb65(6)*abb65(9)
      abb65(28)=-c1*abb65(20)
      abb65(27)=abb65(27)+abb65(28)
      abb65(27)=abb65(2)*abb65(27)
      abb65(26)=abb65(26)+abb65(27)
      abb65(26)=abb65(2)*abb65(26)
      abb65(27)=abb65(6)*abb65(8)
      abb65(20)=-NC*abb65(20)
      abb65(20)=abb65(27)+abb65(20)
      abb65(20)=c2*abb65(20)
      abb65(20)=abb65(20)+abb65(26)
      abb65(20)=mT*abb65(20)
      abb65(20)=abb65(20)+abb65(22)+abb65(24)
      abb65(20)=mT*abb65(20)
      abb65(22)=mH**2*abb65(4)*abb65(5)
      abb65(24)=abb65(22)*spak2l4
      abb65(26)=abb65(24)*abb65(19)
      abb65(27)=-NC*abb65(26)
      abb65(8)=abb65(27)-abb65(8)
      abb65(8)=spbl5k2*abb65(8)
      abb65(27)=abb65(7)**2
      abb65(28)=abb65(27)*abb65(6)
      abb65(15)=-abb65(15)*abb65(28)
      abb65(8)=abb65(15)+abb65(8)
      abb65(8)=c2*abb65(8)
      abb65(15)=-c1*abb65(26)
      abb65(9)=abb65(15)-abb65(9)
      abb65(9)=spbl5k2*abb65(9)
      abb65(10)=-abb65(10)*abb65(28)
      abb65(9)=abb65(10)+abb65(9)
      abb65(9)=abb65(2)*abb65(9)
      abb65(10)=abb65(26)+abb65(14)
      abb65(10)=spbl5k2*abb65(10)
      abb65(14)=spak1l4*abb65(28)
      abb65(10)=abb65(14)+abb65(10)
      abb65(10)=abb65(10)*abb65(25)
      abb65(9)=abb65(10)+abb65(9)
      abb65(9)=abb65(2)*abb65(9)
      abb65(8)=abb65(20)+abb65(8)+abb65(9)
      abb65(8)=abb65(8)*abb65(17)
      abb65(9)=abb65(2)*c1
      abb65(10)=abb65(9)-abb65(25)
      abb65(14)=-abb65(2)*abb65(21)*abb65(10)
      abb65(15)=c2*NC
      abb65(20)=abb65(15)*abb65(21)
      abb65(14)=abb65(14)-abb65(20)
      abb65(20)=-spak1l4*abb65(14)
      abb65(11)=mT*abb65(11)
      abb65(11)=abb65(11)+abb65(20)
      abb65(11)=4.0_ki*abb65(11)*abb65(16)
      abb65(16)=2.0_ki*abb65(16)
      abb65(20)=-abb65(16)*spak1l5*abb65(14)
      abb65(25)=spak1k2*abb65(14)
      abb65(26)=abb65(10)*abb65(2)
      abb65(28)=abb65(26)*abb65(23)
      abb65(29)=abb65(7)*abb65(15)
      abb65(30)=abb65(6)*abb65(29)
      abb65(30)=abb65(30)+abb65(28)
      abb65(30)=mT**2*spak1k2*abb65(30)
      abb65(25)=abb65(30)+abb65(25)
      abb65(25)=abb65(25)*abb65(16)
      abb65(9)=abb65(9)*spbl5k2
      abb65(9)=abb65(9)-abb65(13)
      abb65(9)=abb65(9)*abb65(2)
      abb65(12)=abb65(12)*NC
      abb65(9)=abb65(9)+abb65(12)
      abb65(12)=abb65(16)*mT
      abb65(13)=-abb65(12)*abb65(9)*abb65(19)*spal3l4
      abb65(19)=abb65(16)*spal4l5*abb65(14)
      abb65(30)=-abb65(16)*spal3l4*abb65(14)
      abb65(22)=abb65(22)-1.0_ki
      abb65(14)=-abb65(14)*abb65(22)*spak2l4
      abb65(21)=abb65(21)*abb65(3)
      abb65(22)=abb65(23)*spak2l4
      abb65(21)=abb65(21)+abb65(22)
      abb65(10)=abb65(2)*abb65(21)*abb65(10)
      abb65(21)=abb65(21)*abb65(15)
      abb65(10)=abb65(21)+abb65(10)
      abb65(10)=mT*abb65(10)
      abb65(9)=abb65(9)*abb65(27)*abb65(3)
      abb65(9)=abb65(10)+abb65(9)
      abb65(9)=mT*abb65(9)
      abb65(9)=abb65(9)+abb65(14)
      abb65(9)=abb65(9)*abb65(16)
      abb65(10)=abb65(6)*abb65(3)
      abb65(14)=-abb65(10)*abb65(29)
      abb65(16)=-abb65(3)*abb65(28)
      abb65(15)=abb65(15)+abb65(26)
      abb65(10)=-mT*abb65(10)*abb65(15)
      abb65(10)=abb65(10)+abb65(14)+abb65(16)
      abb65(10)=mT*abb65(10)
      abb65(14)=-abb65(15)*abb65(24)*abb65(6)
      abb65(10)=abb65(10)+abb65(14)
      abb65(10)=abb65(10)*abb65(17)
      abb65(12)=-abb65(12)*abb65(15)*abb65(6)*spal3l4
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h9_qp
