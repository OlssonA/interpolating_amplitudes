module     p0_ubaru_httbar_abbrevd58h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(30), public :: abb58
   complex(ki), public :: R2d58
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
      abb58(1)=sqrt(mT**2)
      abb58(2)=NC**(-1)
      abb58(3)=es12**(-1)
      abb58(4)=spak2l5**(-1)
      abb58(5)=spak2l4**(-1)
      abb58(6)=spak2l3**(-1)
      abb58(7)=spbl3k2**(-1)
      abb58(8)=spbl5k2**(-1)
      abb58(9)=spbl3k2*spak2l3
      abb58(10)=abb58(9)*abb58(4)
      abb58(11)=spak1k2*abb58(5)
      abb58(12)=abb58(10)*abb58(11)
      abb58(13)=spbl4k2*spak1k2
      abb58(14)=abb58(8)*abb58(13)*abb58(4)**2
      abb58(9)=abb58(14)*abb58(9)
      abb58(9)=abb58(12)+abb58(9)
      abb58(12)=mT**2
      abb58(9)=abb58(9)*abb58(12)
      abb58(15)=abb58(7)*abb58(6)*mH**2
      abb58(16)=abb58(15)*spbl5k2*spak1k2
      abb58(17)=spak1l3*spbl5l3
      abb58(18)=abb58(16)+abb58(17)
      abb58(18)=abb58(18)*spbl4k2
      abb58(19)=abb58(11)*spbl5k2
      abb58(20)=abb58(19)*abb58(15)
      abb58(21)=abb58(17)*abb58(5)
      abb58(20)=abb58(21)+abb58(20)
      abb58(21)=spbl5k2*spak2l5
      abb58(22)=abb58(20)*abb58(21)
      abb58(23)=spbl5l4*spak1l5
      abb58(24)=abb58(10)*abb58(23)
      abb58(9)=abb58(9)+abb58(18)+abb58(22)-abb58(24)
      abb58(9)=abb58(9)*mT
      abb58(18)=c2*abb58(9)
      abb58(22)=abb58(13)*abb58(4)
      abb58(19)=abb58(22)+abb58(19)
      abb58(24)=abb58(1)*mT
      abb58(19)=abb58(19)*abb58(24)
      abb58(25)=spak1l3*spbl4l3
      abb58(25)=abb58(25)+abb58(23)
      abb58(25)=abb58(25)*spbl5k2
      abb58(26)=spbl5l4*spak1l4
      abb58(17)=abb58(26)-abb58(17)
      abb58(17)=abb58(17)*spbl4k2
      abb58(17)=abb58(19)-abb58(25)-abb58(17)
      abb58(19)=-abb58(17)*c2
      abb58(25)=-abb58(1)*abb58(19)
      abb58(18)=abb58(18)+abb58(25)
      abb58(25)=abb58(3)*gHT*e*i_*gs**4*TR**2
      abb58(26)=abb58(25)*abb58(1)
      abb58(18)=abb58(18)*abb58(26)
      abb58(9)=-c1*abb58(9)
      abb58(17)=-c1*abb58(17)
      abb58(27)=abb58(1)*abb58(17)
      abb58(9)=abb58(9)+abb58(27)
      abb58(27)=abb58(25)*abb58(2)
      abb58(28)=abb58(27)*abb58(1)
      abb58(9)=abb58(9)*abb58(28)
      abb58(9)=abb58(18)+abb58(9)
      abb58(18)=2.0_ki*abb58(2)
      abb58(9)=abb58(9)*abb58(18)
      abb58(29)=abb58(11)*abb58(4)
      abb58(14)=abb58(29)+abb58(14)
      abb58(12)=2.0_ki*abb58(12)
      abb58(12)=abb58(14)*abb58(12)
      abb58(14)=abb58(22)*abb58(15)
      abb58(22)=abb58(23)*abb58(4)
      abb58(12)=-abb58(12)+abb58(20)-abb58(14)+2.0_ki*abb58(22)
      abb58(14)=abb58(27)*c1
      abb58(20)=abb58(14)*abb58(24)
      abb58(22)=c2*abb58(24)*abb58(25)
      abb58(20)=abb58(20)-abb58(22)
      abb58(12)=-4.0_ki*abb58(2)*abb58(12)*abb58(20)
      abb58(19)=-abb58(19)*abb58(25)
      abb58(17)=abb58(17)*abb58(27)
      abb58(17)=abb58(19)+abb58(17)
      abb58(17)=abb58(17)*abb58(18)
      abb58(19)=abb58(25)*c2
      abb58(14)=abb58(14)-abb58(19)
      abb58(19)=-spbl5l4*abb58(14)
      abb58(22)=abb58(19)*abb58(18)
      abb58(23)=spbl5k2*spak1l3*abb58(22)
      abb58(16)=abb58(16)*abb58(22)
      abb58(24)=spbl4k2*spak1l3
      abb58(22)=abb58(24)*abb58(22)
      abb58(13)=abb58(18)*abb58(13)*abb58(15)*abb58(19)
      abb58(11)=abb58(18)*abb58(11)*spbl3k2*abb58(20)
      abb58(15)=-abb58(20)*abb58(4)
      abb58(19)=-abb58(15)*abb58(24)*abb58(18)
      abb58(21)=abb58(5)*abb58(21)
      abb58(21)=abb58(21)+spbl4k2
      abb58(24)=2.0_ki*mT
      abb58(21)=abb58(21)*abb58(24)
      abb58(24)=c2*abb58(21)
      abb58(25)=3.0_ki*abb58(1)
      abb58(27)=abb58(25)*c2
      abb58(29)=spbl4k2*abb58(27)
      abb58(24)=abb58(24)+abb58(29)
      abb58(24)=abb58(24)*abb58(26)
      abb58(21)=-c1*abb58(21)
      abb58(25)=abb58(25)*c1
      abb58(29)=-spbl4k2*abb58(25)
      abb58(21)=abb58(21)+abb58(29)
      abb58(21)=abb58(21)*abb58(28)
      abb58(21)=abb58(24)+abb58(21)
      abb58(21)=abb58(21)*abb58(18)
      abb58(24)=8.0_ki*abb58(2)
      abb58(20)=-abb58(24)*abb58(5)*abb58(20)
      abb58(14)=abb58(14)*abb58(18)
      abb58(29)=-spbl4k2*abb58(14)
      abb58(10)=abb58(10)*mT
      abb58(30)=-c2*abb58(10)
      abb58(27)=-spbl5k2*abb58(27)
      abb58(27)=abb58(30)+abb58(27)
      abb58(26)=abb58(27)*abb58(26)
      abb58(10)=c1*abb58(10)
      abb58(25)=spbl5k2*abb58(25)
      abb58(10)=abb58(10)+abb58(25)
      abb58(10)=abb58(10)*abb58(28)
      abb58(10)=abb58(26)+abb58(10)
      abb58(10)=abb58(10)*abb58(18)
      abb58(15)=abb58(15)*abb58(24)
      abb58(14)=spbl5k2*abb58(14)
      R2d58=0.0_ki
      rat2 = rat2 + R2d58
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='58' value='", &
          & R2d58, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd58h13_qp
