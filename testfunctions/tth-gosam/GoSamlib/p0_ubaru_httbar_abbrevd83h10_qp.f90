module     p0_ubaru_httbar_abbrevd83h10_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh10_qp
   implicit none
   private
   complex(ki), dimension(12), public :: abb83
   complex(ki), public :: R2d83
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
      abb83(1)=NC**(-1)
      abb83(2)=spbl4k2**(-1)
      abb83(3)=sqrt(mT**2)
      abb83(4)=spak2l4**(-1)
      abb83(5)=c1*abb83(1)**2
      abb83(6)=abb83(3)*gHT
      abb83(7)=TR**2
      abb83(8)=abb83(6)*abb83(5)*abb83(7)
      abb83(9)=-NC+2.0_ki*abb83(1)
      abb83(6)=abb83(9)*abb83(6)*abb83(7)*c2
      abb83(6)=abb83(8)-abb83(6)
      abb83(8)=gs**4*i_*e
      abb83(6)=abb83(6)*abb83(8)
      abb83(10)=-spbl5k1*spak2l5
      abb83(11)=spbl4k1*spak2l4
      abb83(10)=abb83(11)+abb83(10)
      abb83(10)=abb83(6)*abb83(10)
      abb83(9)=abb83(9)*c2
      abb83(5)=abb83(5)-abb83(9)
      abb83(7)=abb83(8)*abb83(7)*spbk2k1*gHT
      abb83(8)=abb83(5)*abb83(7)*mT*abb83(2)
      abb83(9)=abb83(8)*spak2l5
      abb83(11)=-spbl5l4*abb83(9)
      abb83(10)=abb83(11)+abb83(10)
      abb83(10)=2.0_ki*abb83(10)
      abb83(11)=abb83(8)*abb83(3)**2
      abb83(5)=abb83(7)*abb83(5)
      abb83(7)=abb83(4)*mT**3*abb83(2)**2
      abb83(12)=-spbl5k2*abb83(7)*spak2l5*abb83(5)
      abb83(11)=abb83(12)+abb83(11)
      abb83(11)=2.0_ki*abb83(11)
      abb83(5)=4.0_ki*abb83(5)*abb83(7)
      abb83(7)=2.0_ki*abb83(8)
      abb83(6)=4.0_ki*abb83(6)
      abb83(8)=2.0_ki*spbl5k1*abb83(9)
      R2d83=0.0_ki
      rat2 = rat2 + R2d83
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='83' value='", &
          & R2d83, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd83h10_qp
