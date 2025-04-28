module     p2_gg_httbar_abbrevd22h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(22), public :: abb22
   complex(ki), public :: R2d22
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb22(1)=sqrt(mT**2)
      abb22(2)=es12**(-1)
      abb22(3)=es45**(-1)
      abb22(4)=spak2l4**(-1)
      abb22(5)=spak2l5**(-1)
      abb22(6)=c1-c2
      abb22(7)=spbe2e1*spae1e2*gs**4*i_*TR*mT*e*gHT*abb22(3)
      abb22(8)=abb22(6)*abb22(7)*abb22(2)*abb22(1)
      abb22(9)=-abb22(4)*abb22(8)
      abb22(10)=spbl5k1*spak1k2
      abb22(11)=abb22(9)*abb22(10)
      abb22(8)=-abb22(5)*abb22(8)
      abb22(12)=spbl4k1*spak1k2
      abb22(13)=abb22(8)*abb22(12)
      abb22(11)=abb22(11)+abb22(13)
      abb22(6)=-abb22(7)*abb22(6)
      abb22(7)=-abb22(1)*abb22(6)
      abb22(13)=-abb22(4)*abb22(7)
      abb22(14)=spbl5l3*abb22(13)
      abb22(7)=-abb22(5)*abb22(7)
      abb22(15)=spbl4l3*abb22(7)
      abb22(14)=abb22(15)+abb22(14)
      abb22(14)=spak2l3*abb22(14)
      abb22(15)=abb22(9)*spbl5k2
      abb22(16)=abb22(8)*spbl4k2
      abb22(15)=abb22(15)+abb22(16)
      abb22(16)=abb22(15)*spak1k2
      abb22(17)=spbl3k1*spak2l3
      abb22(18)=-abb22(17)*abb22(16)
      abb22(10)=abb22(10)*abb22(4)
      abb22(12)=abb22(12)*abb22(5)
      abb22(10)=abb22(10)+abb22(12)
      abb22(6)=abb22(10)*abb22(6)*abb22(2)*abb22(1)**3
      abb22(10)=spbl3k2*spak2l3
      abb22(12)=-abb22(11)*abb22(10)
      abb22(6)=abb22(12)+2.0_ki*abb22(6)+abb22(18)+abb22(14)
      abb22(12)=2.0_ki*abb22(11)
      abb22(14)=abb22(9)*spbl5l3
      abb22(18)=abb22(8)*spbl4l3
      abb22(14)=abb22(14)+abb22(18)
      abb22(18)=spak2l3*abb22(14)
      abb22(18)=-abb22(12)+abb22(18)
      abb22(18)=2.0_ki*abb22(18)
      abb22(19)=spak1k2*abb22(14)
      abb22(20)=-abb22(9)*abb22(10)
      abb22(13)=-2.0_ki*abb22(13)+abb22(20)
      abb22(20)=-8.0_ki*abb22(9)
      abb22(10)=-abb22(8)*abb22(10)
      abb22(7)=-2.0_ki*abb22(7)+abb22(10)
      abb22(10)=-8.0_ki*abb22(8)
      abb22(15)=spak2l3*abb22(15)
      abb22(14)=-spak1l3*abb22(14)
      abb22(14)=-2.0_ki*abb22(16)+abb22(14)
      abb22(16)=abb22(9)*abb22(17)
      abb22(21)=4.0_ki*abb22(9)
      abb22(17)=abb22(8)*abb22(17)
      abb22(22)=4.0_ki*abb22(8)
      abb22(9)=-spbl5k1*abb22(9)
      abb22(8)=-spbl4k1*abb22(8)
      abb22(8)=abb22(9)+abb22(8)
      abb22(8)=spak2l3*abb22(8)
      R2d22=abb22(11)
      rat2 = rat2 + R2d22
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='22' value='", &
          & R2d22, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd22h12
