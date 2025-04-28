module     p0_ubaru_httbar_abbrevd21h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(27), public :: abb21
   complex(ki), public :: R2d21
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
      abb21(1)=1.0_ki/(-mT**2+es34)
      abb21(2)=1.0_ki/(es34-3.0_ki*sqrt(mT**2)**2)
      abb21(3)=sqrt(mT**2)
      abb21(4)=NC**(-1)
      abb21(5)=es12**(-1)
      abb21(6)=spbl3k2**(-1)
      abb21(7)=spak2l4**(-1)
      abb21(8)=spbl4k2**(-1)
      abb21(9)=1.0_ki/(sqrt(mT**2)**2)
      abb21(10)=spak2l3**(-1)
      abb21(11)=abb21(3)**2
      abb21(12)=abb21(11)*deltaOS
      abb21(13)=abb21(4)*c1
      abb21(14)=abb21(13)*abb21(12)
      abb21(15)=abb21(12)*c2
      abb21(14)=-abb21(15)+abb21(14)
      abb21(14)=abb21(4)*abb21(14)
      abb21(12)=-c1*abb21(12)
      abb21(15)=abb21(15)*NC
      abb21(12)=abb21(14)+abb21(12)+abb21(15)
      abb21(14)=mT**2
      abb21(15)=-es34+3.0_ki*abb21(14)
      abb21(16)=abb21(1)*TR
      abb21(16)=gHT*abb21(16)**2*abb21(5)*gs**4*e*spbl5k1
      abb21(17)=abb21(16)*i_*spbl4l3
      abb21(18)=abb21(17)*spak2l3
      abb21(19)=4.0_ki*abb21(18)
      abb21(12)=abb21(19)*abb21(12)*abb21(2)*abb21(15)
      abb21(15)=deltaOS*abb21(9)
      abb21(20)=abb21(15)*abb21(11)
      abb21(15)=abb21(15)*es34
      abb21(15)=-abb21(20)+abb21(15)+1.0_ki
      abb21(21)=3.0_ki*abb21(11)
      abb21(15)=abb21(15)*abb21(21)
      abb21(21)=mH**2
      abb21(15)=abb21(15)-abb21(21)-abb21(14)
      abb21(13)=abb21(13)-c2
      abb21(13)=abb21(13)*abb21(4)
      abb21(22)=abb21(13)-c1
      abb21(22)=abb21(15)*abb21(22)
      abb21(15)=c2*NC*abb21(15)
      abb21(23)=c2*NC
      abb21(13)=abb21(13)+abb21(23)-c1
      abb21(23)=abb21(14)*abb21(7)
      abb21(24)=abb21(23)*abb21(8)
      abb21(25)=-spak2l3*abb21(13)*abb21(24)*spbl3k2
      abb21(26)=-spbl4l3*spal3l4*abb21(13)
      abb21(15)=abb21(26)+abb21(25)+abb21(15)+abb21(22)
      abb21(15)=spak2l3*abb21(15)
      abb21(21)=abb21(21)*abb21(6)
      abb21(22)=spbl4k2*abb21(21)
      abb21(25)=abb21(22)*spak2l4
      abb21(26)=-abb21(25)*abb21(13)
      abb21(15)=abb21(15)+abb21(26)
      abb21(26)=2.0_ki*abb21(17)
      abb21(15)=abb21(15)*abb21(26)
      abb21(18)=12.0_ki*abb21(18)*abb21(20)*abb21(13)
      abb21(27)=-1.0_ki+3.0_ki*abb21(20)
      abb21(19)=abb21(19)*abb21(27)*abb21(13)
      abb21(21)=abb21(13)*abb21(20)*abb21(21)
      abb21(20)=spak2l3*abb21(13)*abb21(24)*abb21(20)
      abb21(20)=abb21(20)+abb21(21)
      abb21(17)=12.0_ki*abb21(20)*abb21(17)
      abb21(20)=-abb21(26)*spak2l4*abb21(13)
      abb21(21)=abb21(25)*abb21(10)
      abb21(11)=-abb21(11)+abb21(21)+abb21(14)
      abb21(14)=2.0_ki*abb21(16)
      abb21(14)=abb21(14)*i_
      abb21(11)=-abb21(14)*abb21(11)*abb21(13)
      abb21(16)=-abb21(22)*abb21(13)
      abb21(13)=-spak2l3*abb21(23)*abb21(13)
      abb21(13)=abb21(13)+abb21(16)
      abb21(13)=abb21(13)*abb21(14)
      R2d21=abb21(12)
      rat2 = rat2 + R2d21
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='21' value='", &
          & R2d21, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd21h14_qp
