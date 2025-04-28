module     p0_ubaru_httbar_abbrevd71h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(16), public :: abb71
   complex(ki), public :: R2d71
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
      abb71(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb71(2)=NC**(-1)
      abb71(3)=es12**(-1)
      abb71(4)=sqrt(mT**2)
      abb71(5)=spbl3k2**(-1)
      abb71(6)=spak2l5**(-1)
      abb71(7)=abb71(2)**2
      abb71(7)=abb71(7)-1.0_ki
      abb71(8)=TR**2*gs**4*i_*spbl4k1*e*gHT*abb71(3)*abb71(1)
      abb71(7)=abb71(7)*abb71(8)
      abb71(9)=c1*spak2l3*abb71(7)
      abb71(10)=spbl5l3*abb71(9)
      abb71(8)=abb71(8)*c2
      abb71(11)=spak2l3*abb71(8)
      abb71(12)=abb71(11)*spbl5l3
      abb71(13)=abb71(12)*NC
      abb71(12)=abb71(12)*abb71(2)
      abb71(10)=abb71(13)+abb71(10)-abb71(12)
      abb71(12)=-2.0_ki*abb71(10)
      abb71(13)=mT*abb71(4)
      abb71(14)=abb71(4)**2
      abb71(15)=abb71(13)+abb71(14)
      abb71(15)=2.0_ki*abb71(15)*abb71(10)
      abb71(10)=4.0_ki*abb71(10)
      abb71(7)=c1*abb71(7)
      abb71(16)=abb71(8)*NC
      abb71(8)=abb71(2)*abb71(8)
      abb71(7)=abb71(16)+abb71(7)-abb71(8)
      abb71(8)=abb71(7)*mH**2
      abb71(14)=abb71(13)-abb71(14)
      abb71(7)=-abb71(14)*abb71(7)
      abb71(7)=2.0_ki*abb71(7)-abb71(8)
      abb71(7)=2.0_ki*abb71(7)
      abb71(14)=abb71(11)*abb71(2)
      abb71(9)=abb71(14)-abb71(9)
      abb71(9)=mT*abb71(4)*abb71(9)
      abb71(11)=-abb71(13)*abb71(11)*NC
      abb71(9)=abb71(9)+abb71(11)
      abb71(9)=abb71(6)*abb71(9)
      abb71(8)=spbl5k2*abb71(5)*abb71(8)
      abb71(8)=2.0_ki*abb71(9)+abb71(8)
      abb71(8)=2.0_ki*abb71(8)
      R2d71=abb71(12)
      rat2 = rat2 + R2d71
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='71' value='", &
          & R2d71, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd71h14_qp
