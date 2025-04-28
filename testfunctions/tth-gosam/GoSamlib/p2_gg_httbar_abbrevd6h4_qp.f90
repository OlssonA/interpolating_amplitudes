module     p2_gg_httbar_abbrevd6h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(15), public :: abb6
   complex(ki), public :: R2d6
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb6(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb6(2)=es12**(-1)
      abb6(3)=spbl5k2**(-1)
      abb6(4)=spak2l4**(-1)
      abb6(5)=spak2l3**(-1)
      abb6(6)=spbl3k2**(-1)
      abb6(7)=sqrt(mT**2)
      abb6(8)=1.0_ki/(-mT**2+es34)
      abb6(9)=c2-c1
      abb6(10)=spak1l5*spbl4k1
      abb6(11)=abb6(9)*abb6(10)
      abb6(12)=abb6(9)*abb6(4)
      abb6(13)=abb6(12)*spbl3k1
      abb6(14)=-spal3l5*spak1k2*abb6(13)
      abb6(15)=abb6(3)*spbl3k2*spbl4k1*abb6(9)*spak1l3
      abb6(14)=abb6(15)+abb6(14)+abb6(11)
      abb6(14)=abb6(1)*abb6(14)
      abb6(13)=spak2l3*spak1l5*abb6(13)
      abb6(11)=abb6(11)+abb6(13)
      abb6(11)=abb6(8)*abb6(11)
      abb6(13)=abb6(3)*abb6(8)
      abb6(15)=-spak1l3*spbl4l3*spbk2k1*abb6(9)*abb6(13)
      abb6(11)=abb6(14)+abb6(15)+abb6(11)
      abb6(11)=abb6(2)*abb6(11)
      abb6(14)=abb6(5)*mH**2
      abb6(13)=-abb6(13)*abb6(6)*spbl4k2*abb6(9)*abb6(14)
      abb6(12)=-abb6(1)*spak2l5*abb6(12)*abb6(14)*abb6(6)
      abb6(14)=-abb6(7)*abb6(9)
      abb6(9)=-mT*abb6(9)
      abb6(9)=abb6(9)+abb6(14)
      abb6(15)=abb6(1)+abb6(8)
      abb6(9)=mT*abb6(9)*abb6(15)*abb6(3)*abb6(4)
      abb6(9)=abb6(9)+abb6(11)+abb6(13)+abb6(12)
      abb6(9)=mT*abb6(9)
      abb6(10)=-abb6(2)*abb6(15)*abb6(10)*abb6(14)
      abb6(9)=abb6(10)+abb6(9)
      abb6(9)=9.0_ki/8.0_ki*gHT*e*spbe2e1*spae1e2*NC*TR*i_*abb6(9)*gs**4
      R2d6=0.0_ki
      rat2 = rat2 + R2d6
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='6' value='", &
          & R2d6, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd6h4_qp
