module     p2_gg_httbar_abbrevd101h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(49), public :: abb101
   complex(ki), public :: R2d101
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
      abb101(1)=sqrt(mT**2)
      abb101(2)=es45**(-1)
      abb101(3)=spbl4k2**(-1)
      abb101(4)=spbl5k2**(-1)
      abb101(5)=spak2l3**(-1)
      abb101(6)=spbl3k2**(-1)
      abb101(7)=c1-c2
      abb101(8)=gs**4*i_*TR*mT*e*gHT*abb101(2)
      abb101(9)=abb101(8)*abb101(7)*abb101(1)**3
      abb101(10)=-abb101(3)*abb101(9)
      abb101(11)=spae1e2*spbk2e2
      abb101(12)=abb101(10)*abb101(11)
      abb101(13)=spbk2e1*abb101(12)*spak2l5
      abb101(9)=-abb101(4)*abb101(9)
      abb101(14)=abb101(9)*spbk2e1
      abb101(15)=abb101(11)*spak2l4
      abb101(16)=abb101(14)*abb101(15)
      abb101(13)=abb101(13)+abb101(16)
      abb101(16)=mH**2*abb101(6)*abb101(5)
      abb101(17)=abb101(16)-1.0_ki
      abb101(13)=abb101(13)*abb101(17)
      abb101(17)=2.0_ki*abb101(12)
      abb101(18)=-spak1l5*abb101(17)
      abb101(19)=abb101(9)*abb101(11)
      abb101(20)=2.0_ki*abb101(19)
      abb101(21)=-spak1l4*abb101(20)
      abb101(18)=abb101(21)+abb101(18)
      abb101(18)=spbk1e1*abb101(18)
      abb101(21)=abb101(10)*spae2l5
      abb101(22)=abb101(9)*spae2l4
      abb101(21)=abb101(21)+abb101(22)
      abb101(22)=-2.0_ki*abb101(21)
      abb101(23)=spbk2k1*spae1k1
      abb101(24)=abb101(23)*spbe2e1
      abb101(22)=abb101(24)*abb101(22)
      abb101(25)=spbe2e1*abb101(21)
      abb101(26)=spae1l3*spbl3k2
      abb101(27)=-abb101(25)*abb101(26)
      abb101(7)=abb101(7)*abb101(8)*abb101(1)
      abb101(8)=-abb101(3)*abb101(7)
      abb101(28)=abb101(8)*spak2l5
      abb101(11)=abb101(28)*abb101(11)
      abb101(29)=abb101(11)*spbk1e1
      abb101(30)=spbl3k2*abb101(29)
      abb101(7)=-abb101(4)*abb101(7)
      abb101(31)=abb101(7)*spbl3k2
      abb101(32)=abb101(31)*spbk1e1
      abb101(33)=abb101(15)*abb101(32)
      abb101(30)=abb101(30)+abb101(33)
      abb101(30)=spak1l3*abb101(30)
      abb101(12)=spal3l5*abb101(12)
      abb101(19)=spal3l4*abb101(19)
      abb101(12)=abb101(12)+abb101(19)
      abb101(12)=spbl3e1*abb101(12)
      abb101(12)=abb101(12)+abb101(30)+abb101(27)+abb101(22)+abb101(18)+abb101(&
      &13)
      abb101(13)=-spbk2e1*abb101(11)
      abb101(18)=abb101(7)*spbk2e1
      abb101(19)=-abb101(15)*abb101(18)
      abb101(13)=abb101(13)+abb101(19)
      abb101(19)=abb101(7)*spak2l4
      abb101(19)=abb101(19)+abb101(28)
      abb101(22)=spbk2e2*abb101(19)
      abb101(27)=abb101(16)-2.0_ki
      abb101(27)=abb101(22)*abb101(27)
      abb101(28)=abb101(23)*abb101(27)
      abb101(9)=abb101(9)*spae1l4
      abb101(30)=abb101(10)*spae1l5
      abb101(9)=abb101(9)+abb101(30)
      abb101(30)=spbk2e2*abb101(9)
      abb101(22)=-abb101(22)*abb101(26)
      abb101(26)=abb101(8)*spal3l5
      abb101(33)=abb101(7)*spal3l4
      abb101(26)=abb101(26)+abb101(33)
      abb101(33)=abb101(26)*spbl3k1
      abb101(34)=spae1k1*spbk2e2
      abb101(35)=abb101(34)*abb101(33)
      abb101(22)=abb101(35)+abb101(22)+abb101(30)+abb101(28)
      abb101(28)=abb101(8)*spbk2e2
      abb101(30)=spae1l5*abb101(28)
      abb101(35)=abb101(7)*spbk2e2
      abb101(36)=spae1l4*abb101(35)
      abb101(30)=abb101(30)+abb101(36)
      abb101(21)=spbk2e1*abb101(21)
      abb101(36)=abb101(8)*spbl3k2
      abb101(37)=abb101(36)*spae2l5
      abb101(38)=-spbk1e1*abb101(37)
      abb101(39)=-spae2l4*abb101(32)
      abb101(38)=abb101(38)+abb101(39)
      abb101(38)=spak1l3*abb101(38)
      abb101(21)=abb101(38)+abb101(21)
      abb101(38)=abb101(8)*spae2l5
      abb101(39)=spbk2e1*abb101(38)
      abb101(40)=spae2l4*abb101(18)
      abb101(39)=abb101(39)+abb101(40)
      abb101(40)=spbk2e1*spae1e2
      abb101(10)=abb101(10)*abb101(40)
      abb101(41)=abb101(36)*spae1e2
      abb101(42)=-spak1l3*spbk1e1*abb101(41)
      abb101(10)=abb101(10)+abb101(42)
      abb101(40)=abb101(8)*abb101(40)
      abb101(23)=2.0_ki*abb101(23)
      abb101(42)=abb101(8)*abb101(23)
      abb101(36)=spae1l3*abb101(36)
      abb101(36)=abb101(42)+abb101(36)
      abb101(32)=-spak1l3*abb101(32)
      abb101(14)=abb101(14)+abb101(32)
      abb101(14)=spae1e2*abb101(14)
      abb101(18)=spae1e2*abb101(18)
      abb101(23)=abb101(7)*abb101(23)
      abb101(32)=spae1l3*abb101(31)
      abb101(23)=abb101(23)+abb101(32)
      abb101(32)=spal3l5*abb101(28)
      abb101(42)=spal3l4*abb101(35)
      abb101(32)=abb101(32)+abb101(42)
      abb101(42)=abb101(11)*spbl3k2
      abb101(43)=abb101(15)*abb101(31)
      abb101(42)=abb101(42)+abb101(43)
      abb101(43)=-spae2l4*abb101(31)
      abb101(37)=-abb101(37)+abb101(43)
      abb101(31)=-spae1e2*abb101(31)
      abb101(9)=spbe2e1*abb101(9)
      abb101(16)=abb101(19)*abb101(16)
      abb101(19)=abb101(24)*abb101(16)
      abb101(24)=spae1k1*spbe2e1
      abb101(33)=abb101(24)*abb101(33)
      abb101(9)=abb101(33)+abb101(19)+abb101(9)
      abb101(19)=spae1l5*abb101(8)
      abb101(33)=spae1l4*abb101(7)
      abb101(19)=abb101(19)+abb101(33)
      abb101(19)=spbe2e1*abb101(19)
      abb101(33)=abb101(8)*spbk1e1
      abb101(43)=spak1l5*abb101(33)
      abb101(44)=abb101(7)*spbk1e1
      abb101(45)=spak1l4*abb101(44)
      abb101(43)=abb101(43)+abb101(45)
      abb101(45)=-spbk2e1*abb101(16)
      abb101(46)=-spbl3e1*abb101(26)
      abb101(43)=abb101(45)+abb101(46)+2.0_ki*abb101(43)
      abb101(45)=2.0_ki*abb101(8)
      abb101(46)=2.0_ki*abb101(7)
      abb101(26)=spbe2e1*abb101(26)
      abb101(25)=-2.0_ki*abb101(25)
      abb101(16)=spbe2e1*abb101(16)
      abb101(28)=-spak1l5*abb101(28)
      abb101(35)=-spak1l4*abb101(35)
      abb101(28)=abb101(28)+abb101(35)
      abb101(35)=-spak1l5*abb101(8)
      abb101(47)=-spak1l4*abb101(7)
      abb101(35)=abb101(35)+abb101(47)
      abb101(35)=spbe2e1*abb101(35)
      abb101(11)=spbk2k1*abb101(11)
      abb101(47)=abb101(7)*spbk2k1
      abb101(48)=abb101(15)*abb101(47)
      abb101(11)=abb101(11)+abb101(48)
      abb101(38)=-spbk2k1*abb101(38)
      abb101(48)=-spae2l4*abb101(47)
      abb101(38)=abb101(38)+abb101(48)
      abb101(48)=-spae1e2*abb101(8)*spbk2k1
      abb101(47)=-spae1e2*abb101(47)
      abb101(49)=-abb101(8)*abb101(34)
      abb101(8)=-abb101(8)*abb101(24)
      abb101(34)=-abb101(7)*abb101(34)
      abb101(7)=-abb101(7)*abb101(24)
      abb101(15)=abb101(15)*abb101(44)
      abb101(15)=abb101(29)+abb101(15)
      abb101(24)=-spae2l5*abb101(33)
      abb101(29)=-spae2l4*abb101(44)
      abb101(24)=abb101(24)+abb101(29)
      abb101(29)=-spae1e2*abb101(33)
      abb101(33)=-spae1e2*abb101(44)
      R2d101=0.0_ki
      rat2 = rat2 + R2d101
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='101' value='", &
          & R2d101, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd101h0_qp
