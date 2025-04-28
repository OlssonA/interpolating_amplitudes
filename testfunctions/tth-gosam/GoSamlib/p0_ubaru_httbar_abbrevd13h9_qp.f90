module     p0_ubaru_httbar_abbrevd13h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
   implicit none
   private
   complex(ki), dimension(23), public :: abb13
   complex(ki), public :: R2d13
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
      abb13(1)=sqrt(mT**2)
      abb13(2)=NC**(-1)
      abb13(3)=es12**(-1)
      abb13(4)=es45**(-1)
      abb13(5)=spak2l3**(-1)
      abb13(6)=spbl3k2**(-1)
      abb13(7)=spbl4k2**(-1)
      abb13(8)=spak2l5**(-1)
      abb13(9)=c1*abb13(2)
      abb13(9)=abb13(9)-c2
      abb13(9)=abb13(9)*i_*e*gHT*abb13(4)*TR**2*gs**4
      abb13(10)=abb13(9)*abb13(3)
      abb13(11)=-abb13(1)*abb13(10)
      abb13(12)=spak1l4*spbl5k2
      abb13(13)=abb13(11)*abb13(12)
      abb13(14)=-4.0_ki*abb13(13)
      abb13(9)=abb13(9)*abb13(1)
      abb13(15)=mH**2*abb13(6)*abb13(5)
      abb13(16)=-abb13(9)*abb13(15)
      abb13(10)=abb13(1)**3*abb13(10)
      abb13(10)=2.0_ki*abb13(10)+abb13(16)
      abb13(10)=abb13(12)*abb13(10)
      abb13(16)=spbl5l3*spak1l3
      abb13(17)=spbk2k1*spak1l4
      abb13(18)=abb13(17)*abb13(16)
      abb13(19)=spal3l4*spbl3k2
      abb13(20)=-abb13(19)*spak1k2*spbl5k2
      abb13(18)=abb13(18)+abb13(20)
      abb13(18)=abb13(11)*abb13(18)
      abb13(20)=abb13(7)*abb13(8)*mT**2
      abb13(21)=abb13(20)*spbl3k2
      abb13(22)=abb13(21)*spak1l3
      abb13(23)=-abb13(9)*abb13(22)
      abb13(10)=abb13(23)+abb13(18)+abb13(10)
      abb13(10)=4.0_ki*abb13(10)
      abb13(12)=-abb13(12)*abb13(15)
      abb13(12)=abb13(12)+abb13(22)
      abb13(12)=8.0_ki*abb13(11)*abb13(12)
      abb13(13)=8.0_ki*abb13(13)
      abb13(18)=abb13(11)*spbl5k2
      abb13(22)=2.0_ki+abb13(15)
      abb13(22)=spak1k2*abb13(22)*abb13(18)
      abb13(16)=abb13(11)*abb13(16)
      abb13(16)=abb13(22)+abb13(16)
      abb13(16)=4.0_ki*abb13(16)
      abb13(22)=spbl5l3*spak1l4
      abb13(21)=-spak1k2*abb13(21)
      abb13(21)=abb13(21)-abb13(22)
      abb13(22)=4.0_ki*abb13(11)
      abb13(21)=abb13(21)*abb13(22)
      abb13(17)=abb13(19)-2.0_ki*abb13(17)
      abb13(17)=abb13(17)*abb13(22)
      abb13(19)=-4.0_ki*spal3l4*abb13(18)
      abb13(22)=spak2l3*spbl3k2*abb13(11)
      abb13(9)=abb13(22)-2.0_ki*abb13(9)
      abb13(9)=abb13(20)*abb13(9)
      abb13(15)=-spak2l4*abb13(15)*abb13(18)
      abb13(9)=abb13(15)+abb13(9)
      abb13(9)=4.0_ki*abb13(9)
      abb13(15)=32.0_ki*abb13(11)*abb13(20)
      abb13(11)=-16.0_ki*abb13(11)
      R2d13=abb13(14)
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd13h9_qp
