module     p2_gg_httbar_abbrevd253h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
   implicit none
   private
   complex(ki), dimension(61), public :: abb253
   complex(ki), public :: R2d253
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
      abb253(1)=sqrt(mT**2)
      abb253(2)=NC**(-1)
      abb253(3)=spak2l4**(-1)
      abb253(4)=spak2l3**(-1)
      abb253(5)=spbl3k2**(-1)
      abb253(6)=spak2l5**(-1)
      abb253(7)=mH**2*abb253(5)*abb253(4)
      abb253(8)=c2*abb253(2)
      abb253(8)=abb253(8)-c3
      abb253(8)=abb253(8)*gs**4*i_*TR*e*gHT
      abb253(9)=-abb253(8)*abb253(1)**2
      abb253(10)=abb253(7)*abb253(9)
      abb253(11)=spbl4k1*spae1k1
      abb253(12)=abb253(11)*abb253(10)
      abb253(13)=-abb253(8)*abb253(1)**3
      abb253(14)=abb253(3)*mT
      abb253(15)=abb253(13)*abb253(14)
      abb253(16)=spae1k2*abb253(15)
      abb253(12)=abb253(16)+abb253(12)
      abb253(16)=spae2k2*spbe2e1
      abb253(17)=abb253(16)*spbl5k2
      abb253(12)=abb253(17)*abb253(12)
      abb253(18)=spae1l3*spbl4l3
      abb253(18)=-abb253(18)+2.0_ki*abb253(11)
      abb253(19)=abb253(6)*mT
      abb253(20)=abb253(13)*abb253(19)
      abb253(21)=abb253(20)*abb253(16)
      abb253(22)=abb253(21)*abb253(18)
      abb253(23)=abb253(9)*spbk2e2
      abb253(24)=spbl4e1*spae1e2
      abb253(25)=abb253(23)*abb253(24)
      abb253(26)=spbl5l3*abb253(25)
      abb253(27)=abb253(14)*spbl5e2
      abb253(28)=abb253(13)*abb253(27)
      abb253(29)=abb253(28)*spae1e2
      abb253(30)=spbl3e1*abb253(29)
      abb253(31)=abb253(9)*spbl5e1
      abb253(32)=abb253(31)*spae1e2
      abb253(33)=abb253(32)*spbk2e2
      abb253(34)=-spbl4l3*abb253(33)
      abb253(26)=abb253(30)+abb253(26)+abb253(34)
      abb253(26)=spak2l3*abb253(26)
      abb253(30)=abb253(10)*abb253(16)
      abb253(34)=spbl5k1*spae1k1
      abb253(35)=-abb253(34)*abb253(30)
      abb253(36)=-spae1k2*abb253(7)*abb253(21)
      abb253(35)=abb253(35)+abb253(36)
      abb253(35)=spbl4k2*abb253(35)
      abb253(36)=spbl5e2*abb253(24)
      abb253(37)=-spbl4e2*spbl5e1*spae1e2
      abb253(36)=abb253(37)+abb253(36)
      abb253(36)=abb253(36)*abb253(8)*abb253(1)**4
      abb253(37)=spbl5k1*abb253(25)
      abb253(38)=-spbl4k1*abb253(33)
      abb253(37)=abb253(38)+abb253(37)
      abb253(37)=spak1k2*abb253(37)
      abb253(29)=2.0_ki*abb253(29)
      abb253(38)=spbk1e1*spak1k2
      abb253(39)=abb253(38)*abb253(29)
      abb253(40)=abb253(11)*spbl5l3
      abb253(41)=abb253(34)*spbl4l3
      abb253(40)=abb253(40)-abb253(41)
      abb253(41)=abb253(9)*spbe2e1
      abb253(42)=spae2l3*abb253(41)*abb253(40)
      abb253(43)=-abb253(1)*abb253(8)
      abb253(44)=abb253(43)*abb253(14)*spak2l3
      abb253(45)=spbl5k2*abb253(44)
      abb253(16)=abb253(45)*abb253(16)
      abb253(46)=spbl3k1*spae1k1
      abb253(47)=-abb253(46)*abb253(16)
      abb253(12)=abb253(47)+abb253(42)+abb253(39)+2.0_ki*abb253(37)+abb253(35)+&
      &abb253(26)+abb253(22)+abb253(12)+abb253(36)
      abb253(22)=abb253(24)*abb253(9)
      abb253(26)=spbl5e2*abb253(22)
      abb253(35)=spbl4e2*abb253(32)
      abb253(26)=abb253(26)-abb253(35)
      abb253(14)=abb253(43)*abb253(14)
      abb253(35)=abb253(17)*abb253(14)
      abb253(36)=spae1k2*abb253(35)
      abb253(26)=abb253(36)-3.0_ki*abb253(26)
      abb253(36)=abb253(8)*spbk2e2
      abb253(37)=abb253(36)*spak2l3
      abb253(39)=-abb253(37)*abb253(40)
      abb253(28)=-spae1k2*abb253(28)
      abb253(27)=abb253(27)*abb253(43)
      abb253(40)=abb253(27)*spak2l3
      abb253(42)=abb253(46)*abb253(40)
      abb253(28)=abb253(42)+abb253(28)+abb253(39)
      abb253(39)=-spae1k2*abb253(27)
      abb253(42)=abb253(8)*spae1k1
      abb253(47)=abb253(42)*spbl4k1
      abb253(48)=spbl5e2*abb253(47)
      abb253(42)=abb253(42)*spbl5k1
      abb253(49)=-spbl4e2*abb253(42)
      abb253(39)=abb253(49)+abb253(48)+abb253(39)
      abb253(48)=abb253(10)*spbl5k2
      abb253(49)=-spbl4e1*abb253(48)
      abb253(50)=abb253(7)*spbl4k2
      abb253(51)=abb253(31)*abb253(50)
      abb253(45)=spbl3e1*abb253(45)
      abb253(45)=abb253(45)+abb253(51)+abb253(49)
      abb253(45)=spae2k2*abb253(45)
      abb253(49)=abb253(19)*spae2k2
      abb253(51)=abb253(43)*abb253(49)
      abb253(52)=abb253(51)*abb253(50)
      abb253(53)=2.0_ki*abb253(14)
      abb253(54)=spae2k2*abb253(53)*spbl5k2
      abb253(52)=abb253(52)-abb253(54)
      abb253(54)=-abb253(52)*abb253(38)
      abb253(13)=-spbl4e1*abb253(13)*abb253(49)
      abb253(49)=spbl4l3*abb253(31)
      abb253(9)=abb253(9)*spbl4e1
      abb253(55)=-spbl5l3*abb253(9)
      abb253(49)=abb253(49)+abb253(55)
      abb253(49)=spae2l3*abb253(49)
      abb253(55)=spbl4k1*abb253(31)
      abb253(56)=-spbl5k1*abb253(9)
      abb253(55)=abb253(55)+abb253(56)
      abb253(55)=spak1e2*abb253(55)
      abb253(56)=abb253(51)*spbk1e1
      abb253(57)=spak1l3*spbl4l3
      abb253(58)=-abb253(56)*abb253(57)
      abb253(13)=abb253(58)+2.0_ki*abb253(55)+abb253(49)+abb253(54)+abb253(13)+&
      &abb253(45)
      abb253(45)=-spbl4e1*abb253(51)
      abb253(49)=-abb253(11)*abb253(41)
      abb253(15)=-spbe2e1*abb253(15)
      abb253(23)=spbl4e1*abb253(23)
      abb253(15)=abb253(15)+abb253(23)
      abb253(15)=spae1k2*abb253(15)
      abb253(23)=abb253(44)*spbe2e1
      abb253(46)=abb253(46)*abb253(23)
      abb253(15)=abb253(46)+abb253(49)+abb253(15)
      abb253(14)=abb253(14)*spbe2e1
      abb253(46)=-spbl4e1*abb253(36)
      abb253(46)=-abb253(14)+abb253(46)
      abb253(46)=spae1k2*abb253(46)
      abb253(49)=-abb253(38)*abb253(53)
      abb253(44)=-spbl3e1*abb253(44)
      abb253(9)=abb253(44)+3.0_ki*abb253(9)+abb253(49)
      abb253(44)=-spbl4e1*abb253(8)
      abb253(49)=-spae2l3*spbl4l3
      abb253(54)=-spak1e2*spbl4k1
      abb253(49)=abb253(54)+abb253(49)
      abb253(49)=abb253(41)*abb253(49)
      abb253(30)=-spbl4k2*abb253(30)
      abb253(30)=abb253(30)+abb253(49)
      abb253(49)=spbl4l3*abb253(37)
      abb253(54)=abb253(36)*spak1k2
      abb253(55)=spbl4k1*abb253(54)
      abb253(49)=abb253(49)+abb253(55)
      abb253(55)=abb253(41)*abb253(34)
      abb253(58)=-spae1k2*abb253(31)*spbk2e2
      abb253(55)=abb253(55)+abb253(58)
      abb253(58)=spae1k2*spbl5e1*abb253(36)
      abb253(31)=-3.0_ki*abb253(31)
      abb253(59)=spbl5e1*abb253(8)
      abb253(60)=spae2l3*spbl5l3
      abb253(61)=spak1e2*spbl5k1
      abb253(60)=abb253(61)+abb253(60)
      abb253(41)=abb253(41)*abb253(60)
      abb253(10)=abb253(10)*abb253(17)
      abb253(10)=2.0_ki*abb253(21)+abb253(10)+abb253(41)
      abb253(17)=-spbl5k1*abb253(54)
      abb253(21)=-spbl5l3*abb253(37)
      abb253(17)=abb253(17)+abb253(21)
      abb253(21)=-spbl5l3*abb253(22)
      abb253(37)=spbl4l3*abb253(32)
      abb253(21)=abb253(37)+abb253(21)
      abb253(37)=-spbl4l3*abb253(42)
      abb253(41)=spbl5l3*abb253(47)
      abb253(37)=abb253(37)+abb253(41)
      abb253(41)=-spbl4l3*abb253(8)
      abb253(54)=spbl5l3*abb253(8)
      abb253(60)=abb253(51)*spbl4l3
      abb253(20)=-abb253(20)-abb253(48)
      abb253(20)=abb253(24)*abb253(20)
      abb253(19)=abb253(43)*abb253(19)
      abb253(43)=abb253(19)*spae1e2
      abb253(38)=-abb253(43)*abb253(38)
      abb253(38)=abb253(38)+abb253(32)
      abb253(38)=abb253(50)*abb253(38)
      abb253(48)=abb253(43)*spbk1e1
      abb253(57)=-abb253(48)*abb253(57)
      abb253(20)=abb253(57)+abb253(38)+abb253(20)
      abb253(24)=-abb253(24)*abb253(19)
      abb253(18)=-abb253(19)*abb253(18)
      abb253(7)=abb253(7)*spbl5k2
      abb253(38)=abb253(47)*abb253(7)
      abb253(47)=spae1k2*abb253(19)
      abb253(42)=-abb253(42)+abb253(47)
      abb253(42)=abb253(42)*abb253(50)
      abb253(18)=abb253(42)+abb253(38)+abb253(18)
      abb253(38)=-abb253(8)*abb253(50)
      abb253(7)=abb253(8)*abb253(7)
      abb253(7)=-2.0_ki*abb253(19)+abb253(7)
      abb253(19)=-spbl4l3*abb253(43)
      abb253(42)=-abb253(50)*abb253(43)
      abb253(22)=spbl5k1*abb253(22)
      abb253(32)=-spbl4k1*abb253(32)
      abb253(22)=abb253(32)+abb253(22)
      abb253(22)=2.0_ki*abb253(22)
      abb253(32)=spbl4k1*abb253(8)
      abb253(8)=-spbl5k1*abb253(8)
      abb253(47)=-spak1k2*abb253(35)
      abb253(50)=spak1k2*abb253(27)
      abb253(57)=spak1k2*abb253(14)
      abb253(51)=spbl4k1*abb253(51)
      abb253(43)=spbl4k1*abb253(43)
      abb253(25)=-2.0_ki*abb253(25)
      abb253(11)=abb253(36)*abb253(11)
      abb253(33)=2.0_ki*abb253(33)
      abb253(34)=-abb253(36)*abb253(34)
      abb253(35)=-spae1k1*abb253(35)
      abb253(27)=spae1k1*abb253(27)
      abb253(14)=spae1k1*abb253(14)
      R2d253=0.0_ki
      rat2 = rat2 + R2d253
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='253' value='", &
          & R2d253, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd253h12_qp
