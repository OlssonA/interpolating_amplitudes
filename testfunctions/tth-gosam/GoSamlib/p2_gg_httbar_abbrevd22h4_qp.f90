module     p2_gg_httbar_abbrevd22h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(27), public :: abb22
   complex(ki), public :: R2d22
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
      abb22(1)=sqrt(mT**2)
      abb22(2)=es12**(-1)
      abb22(3)=es45**(-1)
      abb22(4)=spak2l4**(-1)
      abb22(5)=spbl5k2**(-1)
      abb22(6)=spak2l3**(-1)
      abb22(7)=spbl3k2**(-1)
      abb22(8)=spbl4k1*spak1l5
      abb22(9)=spbl4k2*spak2l5
      abb22(8)=abb22(8)-abb22(9)
      abb22(10)=c1-c2
      abb22(10)=abb22(10)*spae1e2*gs**4*i_*TR*spbe2e1*e*gHT*abb22(3)
      abb22(11)=abb22(10)*abb22(1)
      abb22(12)=-abb22(2)*abb22(11)
      abb22(13)=abb22(8)*abb22(12)
      abb22(14)=mT**2*abb22(5)*abb22(4)
      abb22(15)=abb22(11)*abb22(14)
      abb22(13)=abb22(13)+abb22(15)
      abb22(16)=spak2l3*spak1l5
      abb22(17)=spak1l3*spak2l5
      abb22(16)=abb22(17)+abb22(16)
      abb22(16)=spbk2k1*spbl4l3*abb22(16)
      abb22(17)=spbl3k2*spbl4k1
      abb22(18)=spbl3k1*spbl4k2
      abb22(17)=abb22(18)+abb22(17)
      abb22(17)=spak1k2*spal3l5*abb22(17)
      abb22(16)=abb22(17)+abb22(16)
      abb22(16)=abb22(12)*abb22(16)
      abb22(8)=-abb22(8)*abb22(2)
      abb22(8)=abb22(14)+abb22(8)
      abb22(8)=abb22(10)*abb22(8)*abb22(1)**3
      abb22(10)=abb22(6)*abb22(7)*mH**2
      abb22(9)=abb22(10)*abb22(9)
      abb22(17)=abb22(14)*spak2l3
      abb22(18)=abb22(17)*spbl3k2
      abb22(19)=-abb22(18)-abb22(9)
      abb22(11)=abb22(11)*abb22(19)
      abb22(8)=abb22(11)+abb22(8)
      abb22(8)=2.0_ki*abb22(8)+abb22(16)
      abb22(9)=abb22(18)-abb22(9)
      abb22(9)=abb22(12)*abb22(9)
      abb22(9)=-2.0_ki*abb22(15)+abb22(9)
      abb22(9)=4.0_ki*abb22(9)
      abb22(11)=16.0_ki*abb22(12)*abb22(14)
      abb22(15)=2.0_ki*abb22(13)
      abb22(16)=spak2l3*spbl4l3
      abb22(18)=spak1k2*spbl4k1
      abb22(16)=abb22(16)-2.0_ki*abb22(18)
      abb22(16)=abb22(12)*abb22(16)
      abb22(18)=abb22(10)+2.0_ki
      abb22(18)=abb22(18)*abb22(12)
      abb22(19)=-spak1k2*spbl4k2*abb22(18)
      abb22(20)=abb22(12)*spbl4l3
      abb22(21)=-spak1l3*abb22(20)
      abb22(19)=abb22(19)+abb22(21)
      abb22(20)=-spak2l5*abb22(20)
      abb22(21)=spbl4l3*spak1l5
      abb22(22)=abb22(14)*spbl3k2
      abb22(23)=spak1k2*abb22(22)
      abb22(21)=abb22(21)+abb22(23)
      abb22(21)=abb22(12)*abb22(21)
      abb22(23)=spbl3k2*spal3l5
      abb22(24)=spbk2k1*spak1l5
      abb22(23)=abb22(23)-2.0_ki*abb22(24)
      abb22(23)=abb22(12)*abb22(23)
      abb22(24)=4.0_ki*abb22(12)
      abb22(25)=abb22(12)*spal3l5
      abb22(26)=-spbl4k2*abb22(25)
      abb22(27)=spak1l5*spbl4k2*abb22(10)
      abb22(22)=-spak1l3*abb22(22)
      abb22(22)=abb22(27)+abb22(22)
      abb22(22)=abb22(12)*abb22(22)
      abb22(18)=-spbk2k1*spak2l5*abb22(18)
      abb22(25)=-spbl3k1*abb22(25)
      abb22(18)=abb22(18)+abb22(25)
      abb22(25)=spal3l5*spbl4k1
      abb22(27)=spbk2k1*abb22(17)
      abb22(25)=abb22(25)+abb22(27)
      abb22(25)=abb22(12)*abb22(25)
      abb22(10)=spbl4k1*spak2l5*abb22(10)
      abb22(17)=-spbl3k1*abb22(17)
      abb22(10)=abb22(10)+abb22(17)
      abb22(10)=abb22(12)*abb22(10)
      abb22(12)=-abb22(14)*abb22(24)
      R2d22=abb22(13)
      rat2 = rat2 + R2d22
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='22' value='", &
          & R2d22, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd22h4_qp
