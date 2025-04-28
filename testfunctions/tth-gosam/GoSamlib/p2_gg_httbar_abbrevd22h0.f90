module     p2_gg_httbar_abbrevd22h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
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
      abb22(4)=spbl4k2**(-1)
      abb22(5)=spbl5k2**(-1)
      abb22(6)=c1-c2
      abb22(7)=spbe2e1*spae1e2*gs**4*i_*TR*mT*e*gHT*abb22(3)
      abb22(8)=abb22(6)*abb22(7)*abb22(2)*abb22(1)
      abb22(9)=-abb22(4)*abb22(8)
      abb22(10)=spak1l5*spbk2k1
      abb22(11)=abb22(9)*abb22(10)
      abb22(8)=-abb22(5)*abb22(8)
      abb22(12)=spak1l4*spbk2k1
      abb22(13)=abb22(8)*abb22(12)
      abb22(11)=abb22(11)+abb22(13)
      abb22(6)=-abb22(7)*abb22(6)
      abb22(7)=-abb22(1)*abb22(6)
      abb22(13)=-abb22(4)*abb22(7)
      abb22(14)=-spal3l5*abb22(13)
      abb22(7)=-abb22(5)*abb22(7)
      abb22(15)=-spal3l4*abb22(7)
      abb22(14)=abb22(15)+abb22(14)
      abb22(14)=spbl3k2*abb22(14)
      abb22(15)=abb22(9)*spak2l5
      abb22(16)=abb22(8)*spak2l4
      abb22(15)=abb22(15)+abb22(16)
      abb22(16)=abb22(15)*spbk2k1
      abb22(17)=spak1l3*spbl3k2
      abb22(18)=abb22(17)*abb22(16)
      abb22(10)=-abb22(10)*abb22(4)
      abb22(12)=-abb22(12)*abb22(5)
      abb22(10)=abb22(10)+abb22(12)
      abb22(6)=abb22(10)*abb22(6)*abb22(2)*abb22(1)**3
      abb22(10)=spak2l3*spbl3k2
      abb22(12)=abb22(11)*abb22(10)
      abb22(6)=abb22(12)+2.0_ki*abb22(6)+abb22(18)+abb22(14)
      abb22(12)=2.0_ki*abb22(11)
      abb22(14)=abb22(9)*spal3l5
      abb22(18)=abb22(8)*spal3l4
      abb22(14)=abb22(14)+abb22(18)
      abb22(18)=-spbl3k2*abb22(14)
      abb22(18)=abb22(12)+abb22(18)
      abb22(18)=2.0_ki*abb22(18)
      abb22(19)=abb22(9)*abb22(10)
      abb22(13)=2.0_ki*abb22(13)+abb22(19)
      abb22(19)=8.0_ki*abb22(9)
      abb22(20)=-abb22(9)*abb22(17)
      abb22(10)=abb22(8)*abb22(10)
      abb22(7)=2.0_ki*abb22(7)+abb22(10)
      abb22(10)=8.0_ki*abb22(8)
      abb22(17)=-abb22(8)*abb22(17)
      abb22(15)=-spbl3k2*abb22(15)
      abb22(21)=spak1l5*abb22(9)
      abb22(22)=spak1l4*abb22(8)
      abb22(21)=abb22(21)+abb22(22)
      abb22(21)=spbl3k2*abb22(21)
      abb22(22)=-spbk2k1*abb22(14)
      abb22(14)=spbl3k1*abb22(14)
      abb22(14)=2.0_ki*abb22(16)+abb22(14)
      abb22(9)=-4.0_ki*abb22(9)
      abb22(8)=-4.0_ki*abb22(8)
      R2d22=-abb22(11)
      rat2 = rat2 + R2d22
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='22' value='", &
          & R2d22, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd22h0
